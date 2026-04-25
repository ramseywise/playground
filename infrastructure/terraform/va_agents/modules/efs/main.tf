resource "aws_efs_file_system" "billy" {
  encrypted = true

  tags = { Name = "${var.name_prefix}-billy-efs" }
}

resource "aws_security_group" "efs" {
  name_prefix = "${var.name_prefix}-efs-"
  vpc_id      = var.vpc_id

  ingress {
    description     = "NFS from ECS tasks"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [var.ecs_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.name_prefix}-efs-sg" }

  lifecycle { create_before_destroy = true }
}

resource "aws_efs_mount_target" "billy" {
  count = length(var.subnet_ids)

  file_system_id  = aws_efs_file_system.billy.id
  subnet_id       = var.subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "billy" {
  file_system_id = aws_efs_file_system.billy.id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/billy-data"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "755"
    }
  }

  tags = { Name = "${var.name_prefix}-billy-ap" }
}
